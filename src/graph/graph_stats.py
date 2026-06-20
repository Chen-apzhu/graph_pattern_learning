"""
Graph Statistics — structural statistics and reporting for SchoolGraphData.

Computes standard graph metrics (degree, density, clustering, path length,
diameter) and domain-specific statistics (room type distribution, daylight
compliance rate, acoustic separation rate).
"""

from __future__ import annotations

from typing import Dict, List
import math

import networkx as nx

from utils.enums import RoomType, EnvNodeType
from graph.graph_utils import GraphAnalyzer


class GraphStats:
    """
    Computes and reports structural statistics of a SchoolGraphData.

    Usage:
        sg = SchoolGraphData(hetero_data)
        stats = GraphStats(sg)
        print(stats.report())
    """

    def __init__(self, school_graph):
        """
        Args:
            school_graph: SchoolGraphData instance.
        """
        self.sg = school_graph
        self.analyzer = GraphAnalyzer(school_graph)
        self._nx = self.analyzer.nx_graph

    # ========================================================================
    # General graph metrics
    # ========================================================================

    def summary(self) -> Dict:
        """
        Compute a comprehensive summary of graph statistics.

        Returns dict with keys:
            num_rooms, num_env_nodes,
            num_physical_edges, num_acoustic_edges, num_sight_edges,
            avg_degree_physical, avg_degree_acoustic,
            density_physical, density_acoustic,
            num_connected_components_physical,
            avg_clustering, avg_path_length, diameter,
            graph_is_connected
        """
        edge_counts = self.sg.edge_counts()
        full_G = self._nx
        phys_G = self.analyzer._build_edge_type_subgraph('physical_connects')

        n = self.sg.num_rooms + self.sg.num_env_nodes
        n_rooms = self.sg.num_rooms

        # Average degree (physical)
        if n_rooms > 0:
            total_degree = sum(
                phys_G.degree(f"room_{i}") for i in range(n_rooms)
                if f"room_{i}" in phys_G
            )
            avg_degree_phys = total_degree / n_rooms if n_rooms > 0 else 0.0
        else:
            avg_degree_phys = 0.0

        # Density
        if n_rooms > 1:
            max_edges_phys = n_rooms * (n_rooms - 1) / 2
            phys_rr = edge_counts.get('room→physical_connects→room', 0)
            density_phys = phys_rr / max_edges_phys
        else:
            density_phys = 0.0

        # Connected components
        components_phys = list(nx.connected_components(phys_G))
        n_components_phys = len(components_phys)

        # Clustering, path length, diameter (on full graph)
        try:
            avg_clustering = nx.average_clustering(full_G)
        except ZeroDivisionError:
            avg_clustering = 0.0

        if n_components_phys == 1 and n_rooms > 1:
            try:
                avg_path_length = nx.average_shortest_path_length(phys_G)
                diameter = nx.diameter(phys_G)
            except (nx.NetworkXError, ZeroDivisionError):
                avg_path_length = 0.0
                diameter = 0
        else:
            largest_cc = max(components_phys, key=len) if components_phys else set()
            if len(largest_cc) > 1:
                sub = phys_G.subgraph(largest_cc)
                try:
                    avg_path_length = nx.average_shortest_path_length(sub)
                    diameter = nx.diameter(sub)
                except (nx.NetworkXError, ZeroDivisionError):
                    avg_path_length = 0.0
                    diameter = 0
            else:
                avg_path_length = 0.0
                diameter = 0

        return {
            'num_rooms': self.sg.num_rooms,
            'num_env_nodes': self.sg.num_env_nodes,
            'num_physical_edges': edge_counts.get('room→physical_connects→room', 0) + edge_counts.get('room→physical_connects→environment', 0),
            'num_acoustic_edges': edge_counts.get('room→acoustic_blocks→room', 0),
            'num_sight_edges': (
                edge_counts.get('room→sight_lines→room', 0) +
                edge_counts.get('room→sight_lines→environment', 0)
            ),
            'avg_degree_physical': round(avg_degree_phys, 2),
            'density_physical': round(density_phys, 4),
            'num_connected_components_physical': n_components_phys,
            'avg_clustering': round(avg_clustering, 4),
            'avg_path_length': round(avg_path_length, 2),
            'diameter': diameter,
            'graph_is_connected': n_components_phys <= 1,
        }

    # ========================================================================
    # Domain-specific statistics
    # ========================================================================

    def room_type_distribution(self) -> Dict[str, int]:
        """
        Count rooms by type from one-hot features.

        Returns:
            Dict[room_type_name, count].
        """
        dist: Dict[str, int] = {}
        room_x = self.sg.room_features

        for i in range(self.sg.num_rooms):
            # RoomType one-hot in [0:13]
            rt_idx = room_x[i, :13].argmax().item()
            room_types = list(RoomType)
            if rt_idx < len(room_types):
                name = room_types[rt_idx].value
                dist[name] = dist.get(name, 0) + 1
            else:
                dist['unknown'] = dist.get('unknown', 0) + 1

        return dist

    def floor_distribution(self) -> Dict[str, int]:
        """
        Count rooms by typical floor type (standard floor).

        Returns:
            Dict[typical_floor_type, count].
        """
        dist: Dict[str, int] = {}
        room_x = self.sg.room_features

        # Use zone one-hot + room type to infer typical floor grouping
        # Ground: zone=admin, service mixed | Teaching: zone=teaching, circulation
        for i in range(self.sg.num_rooms):
            zone_idx = room_x[i, 20:26].argmax().item()
            rt_idx = room_x[i, :13].argmax().item()

            # Infer typical floor from room type + floor height
            floor_mid = room_x[i, 19].item() * 4
            if floor_mid < 0.5:
                tf_type = 'ground'
            elif floor_mid > 3.0:
                tf_type = 'top'
            else:
                tf_type = 'teaching'
            dist[tf_type] = dist.get(tf_type, 0) + 1

        return dist

    def daylight_compliance_rate(self) -> float:
        """
        Fraction of high-daylight rooms that have at least one sight_line edge.

        Formula (§4):
            numerator = count of rooms with daylight >= HIGH AND sight_degree >= 1
            denominator = count of rooms with daylight >= HIGH

        Returns:
            Compliance rate in [0, 1]. Returns 1.0 if no high-daylight rooms.
        """
        room_x = self.sg.room_features
        sight_ei = self.sg.sight_room_edges
        sight_env_ei = self.sg.sight_env_edges

        total_high = 0
        compliant = 0

        for i in range(self.sg.num_rooms):
            dl = room_x[i, 16].item() * 4  # denormalize
            if dl < 3.0:  # below HIGH
                continue

            total_high += 1
            # Count sight lines for this room
            sight_degree = 0
            if sight_ei.numel() > 0:
                sight_degree += (sight_ei[0] == i).sum().item()
                sight_degree += (sight_ei[1] == i).sum().item()
            if sight_env_ei.numel() > 0:
                sight_degree += (sight_env_ei[0] == i).sum().item()

            if sight_degree >= 1:
                compliant += 1

        if total_high == 0:
            return 1.0

        return compliant / total_high

    def acoustic_separation_rate(self) -> float:
        """
        Fraction of noisy/quiet pairs that have adequate separation.

        Checks whether pairs with noise_gap >= 2 have either:
          (a) an acoustic_blocks edge, or
          (b) sufficient physical path distance.

        Returns:
            Adequate separation rate in [0, 1]. Returns 1.0 if no critical pairs.
        """
        room_x = self.sg.room_features
        acous_ei = self.sg.acoustic_edges
        phys_ei = self.sg.physical_edges

        # Build physical adjacency for distance check
        phys_adj: Dict[int, set] = {i: set() for i in range(self.sg.num_rooms)}
        if phys_ei.numel() > 0:
            for j in range(phys_ei.shape[1]):
                s, d = phys_ei[0, j].item(), phys_ei[1, j].item()
                phys_adj[s].add(d)
                phys_adj[d].add(s)

        # Build acoustic adjacency
        acous_adj: Dict[int, set] = {i: set() for i in range(self.sg.num_rooms)}
        if acous_ei.numel() > 0:
            for j in range(acous_ei.shape[1]):
                s, d = acous_ei[0, j].item(), acous_ei[1, j].item()
                acous_adj[s].add(d)
                acous_adj[d].add(s)

        def _bfs_dist(src: int, dst: int) -> int:
            """BFS shortest path in physical graph."""
            if src == dst:
                return 0
            visited = {src}
            queue = [(src, 0)]
            while queue:
                node, dist = queue.pop(0)
                for nb in phys_adj.get(node, set()):
                    if nb == dst:
                        return dist + 1
                    if nb not in visited:
                        visited.add(nb)
                        queue.append((nb, dist + 1))
            return 10**9

        total_pairs = 0
        adequate = 0

        for i in range(self.sg.num_rooms):
            for j in range(i + 1, self.sg.num_rooms):
                ni = int(room_x[i, 17].item() * 4)  # noise_level
                nj = int(room_x[j, 17].item() * 4)
                ti = int(room_x[i, 18].item() * 4)  # noise_tolerance
                tj = int(room_x[j, 18].item() * 4)

                gap_ij = ni - tj
                gap_ji = nj - ti

                if gap_ij >= 2 or gap_ji >= 2:
                    total_pairs += 1
                    has_acoustic = j in acous_adj.get(i, set())
                    if has_acoustic:
                        adequate += 1
                    else:
                        path_dist = _bfs_dist(i, j)
                        if path_dist >= 2:
                            adequate += 1

        if total_pairs == 0:
            return 1.0

        return adequate / total_pairs

    # ========================================================================
    # Edge type statistics
    # ========================================================================

    def edge_type_distribution(self) -> Dict[str, int]:
        """Edge counts by type."""
        return self.sg.edge_counts()

    def edge_density_by_type(self) -> Dict[str, float]:
        """
        Compute edge density for each room-room edge type.

        Density = 2 * |E| / (n * (n-1)) for n room nodes.
        """
        n = self.sg.num_rooms
        if n < 2:
            return {}

        max_edges = n * (n - 1) / 2
        counts = self.sg.edge_counts()
        densities = {}

        for edge_key, count in counts.items():
            if edge_key in ('sight_lines',):  # includes room↔env edges
                continue  # Can't compute simple density for heterogeneous edges
            densities[edge_key] = round(count / max_edges, 4)

        return densities

    # ========================================================================
    # Report
    # ========================================================================

    def report(self) -> str:
        """
        Pretty-print a full statistics report.

        Returns:
            Multi-line formatted string.
        """
        s = self.summary()
        rt = self.room_type_distribution()
        floor = self.floor_distribution()

        lines = [
            "=" * 55,
            "  SCHOOL BUILDING GRAPH STATISTICS",
            "=" * 55,
            "",
            "  ── Node Counts ──",
            f"    Rooms:       {s['num_rooms']}",
            f"    Env Nodes:   {s['num_env_nodes']}",
            "",
            "  ── Edge Counts ──",
            f"    Physical:     {s['num_physical_edges']}",
            f"    Acoustic:     {s['num_acoustic_edges']}",
            f"    Sight:        {s['num_sight_edges']}",
            "",
            "  ── Topology ──",
            f"    Avg Degree (phys):       {s['avg_degree_physical']}",
            f"    Density (phys):          {s['density_physical']}",
            f"    Connected Components:    {s['num_connected_components_physical']}",
            f"    Connected:               {s['graph_is_connected']}",
            f"    Avg Clustering:          {s['avg_clustering']}",
            f"    Avg Path Length:         {s['avg_path_length']}",
            f"    Diameter:                {s['diameter']}",
            "",
            "  ── Compliance ──",
            f"    Daylight:     {self.daylight_compliance_rate():.1%}",
            f"    Acoustic:     {self.acoustic_separation_rate():.1%}",
            "",
            "  ── Room Distribution ──",
        ]

        for rtype, count in sorted(rt.items()):
            lines.append(f"    {rtype:20s}: {count:4d}")

        lines.append("")
        lines.append("  ── Typical Floor Distribution ──")
        for tf, count in sorted(floor.items()):
            lines.append(f"    {tf}: {count} rooms")

        lines.append("")
        lines.append("=" * 55)

        return "\n".join(lines)
