"""
Architectural Quality Metrics — 建筑品质指标

Computes 7 continuous, graph-topology-based quality metrics from HeteroData.
These measure architectural excellence BEYOND code compliance — two compliant
designs can score differently on each metric.

Every metric:
  - Takes PyG HeteroData as input (tensor operations only)
  - Returns float in [0, 1]
  - References GB code where applicable
  - Is independently testable

Usage:
    from metrics.quality_metrics import QualityMetrics

    metrics = QualityMetrics.compute_all(hetero_data)
    score = QualityMetrics.aggregate(metrics)
"""

import torch
import math
from typing import Dict, Optional

from utils.constants import (
    DEFAULT_MAX_AREA, DEFAULT_MAX_OCCUPANCY, DEFAULT_MAX_FLOORS,
)

# Room feature column indices (from feature_engineering.py)
COL_RT_START, COL_RT_END = 0, 13     # RoomType one-hot
COL_AREA = 13                          # area, normalized [0,1]
COL_ASPECT = 14                        # aspect_ratio
COL_OCCUPANCY = 15                     # occupancy, normalized
COL_DAYLIGHT = 16                      # daylight_level, ordinal norm
COL_NOISE_LVL = 17                     # noise_level, ordinal norm
COL_NOISE_TOL = 18                     # noise_tolerance, ordinal norm
COL_FLOOR = 19                         # floor, normalized
COL_ZONE_START, COL_ZONE_END = 20, 26  # ZoneType one-hot (6 zones)
COL_FIRE_EXITS = 26                    # fire_exits_min, normalized

CORRIDOR_IDX = 7   # list(RoomType).index(RoomType.CORRIDOR)


class QualityMetrics:
    """Static methods for computing architectural quality from HeteroData."""

    # ─────────────────────────────────────────────────────────────────
    # Metric 1: Daylight Quality (采光质量)
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def daylight_quality(data) -> float:
        """
        Continuous daylight quality score for HIGH-daylight rooms.

        Each HIGH room gets a score based on its sight_degree:
          - 0 connections → 0.0
          - 1 connection  → 0.5 (one path: direct south OR corridor)
          - 2 connections → 0.8 (direct south AND corridor)
          - 3+ connections → 1.0 (multiple redundant daylight paths)

        Reference: GB50099-2011 §5.1 (natural lighting)
        Range: [0, 1]. Higher = better daylight access.
        """
        room_x = data['room'].x
        n_rooms = room_x.shape[0]
        if n_rooms == 0:
            return 1.0

        high_mask = room_x[:, COL_DAYLIGHT] >= 0.75

        degree = torch.zeros(n_rooms, device=room_x.device)
        for et in [('room', 'sight_lines', 'room'), ('room', 'sight_lines', 'environment')]:
            try:
                ei = data[et].edge_index
                if ei.numel() > 0:
                    degree = degree.scatter_add(0, ei[0], torch.ones(ei.shape[1], device=ei.device))
            except (KeyError, AttributeError):
                continue

        n_high = high_mask.sum().item()
        if n_high == 0:
            return 1.0

        # Continuous score per room: 0.0/0.5/0.8/1.0 based on sight degree
        deg_clipped = degree.clamp(0, 3)
        score_map = torch.tensor([0.0, 0.5, 0.8, 1.0], device=room_x.device)
        per_room_scores = score_map[deg_clipped.long()]
        return float(per_room_scores[high_mask].mean().item())

    # ─────────────────────────────────────────────────────────────────
    # Metric 2: Acoustic Comfort (声学舒适度)
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def acoustic_comfort(data) -> float:
        """
        Mean normalized path distance from noisy rooms to quiet-tolerant rooms.

        Reference: GB50099-2011 §7.3 (acoustic design)
        Range: [0, 1]. Higher = better acoustic separation.
        """
        room_x = data['room'].x
        n_rooms = room_x.shape[0]
        if n_rooms == 0:
            return 1.0

        noisy_mask = room_x[:, COL_NOISE_LVL] >= 0.5  # >= MODERATE
        quiet_mask = room_x[:, COL_NOISE_TOL] <= 0.25  # <= QUIET tolerance

        noisy_idx = noisy_mask.nonzero(as_tuple=True)[0]
        quiet_idx = quiet_mask.nonzero(as_tuple=True)[0]

        if len(noisy_idx) == 0 or len(quiet_idx) == 0:
            return 1.0

        # Build physical adjacency for BFS
        try:
            phys_ei = data['room', 'physical_connects', 'room'].edge_index
        except (KeyError, AttributeError):
            return 0.5

        # Compute shortest paths via BFS (CPU, n_rooms is small)
        distances = []
        max_possible = max(1, n_rooms)
        for ni in noisy_idx.tolist():
            # Simple BFS from noisy node
            visited = {ni}
            queue = [(ni, 0)]
            dist_map = {}
            while queue:
                node, dist = queue.pop(0)
                dist_map[node] = dist
                neighbors = phys_ei[1, phys_ei[0] == node].tolist()
                neighbors += phys_ei[0, phys_ei[1] == node].tolist()
                for nb in neighbors:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append((nb, dist + 1))
            for qi in quiet_idx.tolist():
                if qi in dist_map:
                    distances.append(dist_map[qi] / max_possible)
                else:
                    distances.append(1.0)  # Unreachable = max comfort

        if not distances:
            return 1.0
        return float(torch.tensor(distances).mean().item())

    # ─────────────────────────────────────────────────────────────────
    # Metric 3: Circulation Efficiency (交通效率)
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def circulation_efficiency(data) -> float:
        """
        How close the corridor area ratio is to the optimal 18% (GB50099 §8.2.3).

        Formula: 1 - |corridor_ratio - 0.18| / 0.12
        Range: [0, 1]. 1.0 = exactly 18%; 0.0 = below 6% or above 30%.
        """
        room_x = data['room'].x
        areas = room_x[:, COL_AREA] * DEFAULT_MAX_AREA
        total_area = areas.sum().item()
        if total_area <= 0:
            return 0.5

        corridor_areas = room_x[:, CORRIDOR_IDX] * areas
        corr_ratio = corridor_areas.sum().item() / total_area

        return float(max(0.0, 1.0 - abs(corr_ratio - 0.18) / 0.12))

    # ─────────────────────────────────────────────────────────────────
    # Metric 4: Fire Safety Margin (消防安全余量)
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def fire_safety_margin(data) -> float:
        """
        Mean safety margin for high-occupancy rooms.

        Formula: mean(ReLU(degree - min_required + 1) / 3)
        for rooms with occupancy >= threshold (50 persons).

        Reference: GB50016-2014 §5.5
        Range: [0, 1]. Higher = more evacuation redundancy.
        """
        room_x = data['room'].x
        n_rooms = room_x.shape[0]
        if n_rooms == 0:
            return 1.0

        occupancy = room_x[:, COL_OCCUPANCY] * DEFAULT_MAX_OCCUPANCY
        fire_exits_req = torch.clamp((room_x[:, COL_FIRE_EXITS] * 4.0).round(), min=1)
        high_occ_mask = occupancy >= 50.0

        n_high = high_occ_mask.sum().item()
        if n_high == 0:
            return 1.0

        degree = torch.zeros(n_rooms, device=room_x.device)
        try:
            phys_ei = data['room', 'physical_connects', 'room'].edge_index
            if phys_ei.numel() > 0:
                ones = torch.ones(phys_ei.shape[1], device=phys_ei.device)
                degree = degree.scatter_add(0, phys_ei[0], ones)
                degree = degree.scatter_add(0, phys_ei[1], ones)
        except (KeyError, AttributeError):
            pass

        margin = torch.relu(degree - fire_exits_req + 1) / 3.0
        margin = torch.clamp(margin, 0.0, 1.0)
        return float(margin[high_occ_mask].mean().item())

    # ─────────────────────────────────────────────────────────────────
    # Metric 5: Graph Robustness (图鲁棒性)
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def graph_robustness(data) -> float:
        """
        Normalized algebraic connectivity (lambda_2) of the physical graph.

        Formula: min(lambda_2(L_norm) * 5, 1.0)
        Range: [0, 1]. Higher = more robust circulation network.
        """
        try:
            phys_ei = data['room', 'physical_connects', 'room'].edge_index
        except (KeyError, AttributeError):
            return 0.0

        if phys_ei.numel() < 2:
            return 0.0

        n_r = data['room'].num_nodes
        row, col = phys_ei[0], phys_ei[1]
        ones = torch.ones(row.shape[0], device=phys_ei.device)
        adj = torch.sparse_coo_tensor(
            torch.stack([row, col]), ones, (n_r, n_r)
        ).coalesce()
        adj = adj + adj.t().coalesce()

        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)

        idx = adj.indices()
        vals = adj.values()
        norm_vals = vals * deg_inv_sqrt[idx[0]] * deg_inv_sqrt[idx[1]]
        L_norm = torch.sparse_coo_tensor(idx, -norm_vals, (n_r, n_r))
        I_idx = torch.arange(n_r, device=phys_ei.device).unsqueeze(0).repeat(2, 1)
        L_norm = L_norm + torch.sparse_coo_tensor(I_idx, torch.ones(n_r, device=phys_ei.device), (n_r, n_r))
        L_norm = L_norm.coalesce()

        try:
            eigvals = torch.linalg.eigvalsh(L_norm.to_dense())
            lambda_2 = eigvals[1].item() if eigvals.shape[0] > 1 else 0.0
        except Exception:
            lambda_2 = 0.0

        return float(min(lambda_2 * 5.0, 1.0))

    # ─────────────────────────────────────────────────────────────────
    # Metric 6: Space Type Diversity (功能多样性)
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def space_type_diversity(data) -> float:
        """
        Shannon entropy of room type distribution, normalized by ln(13).

        Range: [0, 1]. Higher = more functionally diverse.
        """
        room_x = data['room'].x
        type_counts = room_x[:, :13].sum(dim=0)  # [13]
        total = type_counts.sum().item()
        if total <= 0:
            return 0.0

        probs = type_counts / total
        probs = probs[probs > 0]
        entropy = -(probs * probs.log()).sum().item()
        max_entropy = math.log(13)  # 13 room types
        return float(entropy / max_entropy)

    # ─────────────────────────────────────────────────────────────────
    # Metric 7: Vertical Flow Balance (竖向均衡度)
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def vertical_flow_balance(data) -> float:
        """
        Balance of staircase distribution across floors.

        Formula: 1.0 - CV(stair_count_per_floor)
        Range: [0, 1]. 1.0 = perfectly balanced, 0.0 = highly skewed.
        Reference: GB50016-2014 §5.5
        """
        room_x = data['room'].x
        floor_norm = room_x[:, COL_FLOOR]
        floors = (floor_norm * DEFAULT_MAX_FLOORS).round().long()

        # Staircase index = 9 in room type one-hot
        is_stair = room_x[:, 9] > 0.5

        stair_floors = floors[is_stair]
        unique_floors = torch.unique(floors).tolist()

        if len(unique_floors) <= 1:
            return 1.0

        counts = []
        for f in unique_floors:
            counts.append((stair_floors == f).sum().item())

        counts_t = torch.tensor(counts, dtype=torch.float32)
        mean_c = counts_t.mean().item()
        if mean_c <= 0:
            return 1.0
        cv = counts_t.std().item() / mean_c
        return float(max(0.0, 1.0 - cv))

    # ─────────────────────────────────────────────────────────────────
    # Metric 8: Path Redundancy (路径冗余度)
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def path_redundancy(data) -> float:
        """
        Average edge-disjoint path count between random room pairs.

        Measures how many alternative routes exist between rooms.
        Higher = more resilient circulation (mesh vs tree).

        Approximated by counting how many rooms have degree >= 3
        (indicating branching/looping circulation rather than chain).

        Range: [0, 1]. 0 = all rooms degree <= 2 (tree), 1 = most rooms degree >= 3 (mesh).
        """
        try:
            phys_ei = data['room', 'physical_connects', 'room'].edge_index
        except (KeyError, AttributeError):
            return 0.0

        n_r = data['room'].num_nodes
        if n_r <= 1:
            return 1.0

        degree = torch.zeros(n_r, device=phys_ei.device)
        if phys_ei.numel() > 0:
            ones = torch.ones(phys_ei.shape[1], device=phys_ei.device)
            degree = degree.scatter_add(0, phys_ei[0], ones)
            degree = degree.scatter_add(0, phys_ei[1], ones)

        # Score: proportion of rooms with degree >= 3 (branching circulation)
        branch_ratio = (degree >= 3).float().mean().item()
        return float(min(branch_ratio * 2.0, 1.0))  # scale: 50% branching → 1.0

    # ─────────────────────────────────────────────────────────────────
    # Metric 9: Zone Cohesion (区域凝聚度)
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def zone_cohesion(data) -> float:
        """
        How well rooms of the same functional zone cluster together
        in the physical graph. Uses the ratio of intra-zone edges to
        total edges.

        Range: [0, 1]. Higher = cleaner functional zoning.
        """
        try:
            phys_ei = data['room', 'physical_connects', 'room'].edge_index
        except (KeyError, AttributeError):
            return 0.5

        if phys_ei.numel() < 2:
            return 0.5

        room_x = data['room'].x
        zones = room_x[:, 20:26].argmax(dim=1)  # ZoneType one-hot

        src_z = zones[phys_ei[0]]
        dst_z = zones[phys_ei[1]]
        same_zone = (src_z == dst_z).float().mean().item()

        # Intra-zone ratio: 0 = fully mixed, 1 = fully separated
        # Ideal teaching building: ~0.3-0.5 (some mixing via corridors)
        # Normalize: center at 0.4 with penalty for extremes
        return float(1.0 - abs(same_zone - 0.4) / 0.4)

    # ─────────────────────────────────────────────────────────────────
    # Aggregation
    # ─────────────────────────────────────────────────────────────────

    DEFAULT_WEIGHTS = {
        # daylight_quality: excluded — metric still under development
        'circulation_efficiency': 1.0,
        'fire_safety_margin': 1.0,
        'graph_robustness': 1.0,
        'path_redundancy': 1.0,
        'zone_cohesion': 1.0,
        'space_type_diversity': 0.5,
        'vertical_flow_balance': 0.5,
    }

    @classmethod
    def compute_all(cls, data) -> Dict[str, float]:
        """Compute all active quality metrics from HeteroData.

        Note: acoustic_comfort is computed but excluded from scoring.
        """
        return {
            'daylight_quality': cls.daylight_quality(data),
            'acoustic_comfort': cls.acoustic_comfort(data),  # computed, not scored
            'circulation_efficiency': cls.circulation_efficiency(data),
            'fire_safety_margin': cls.fire_safety_margin(data),
            'graph_robustness': cls.graph_robustness(data),
            'path_redundancy': cls.path_redundancy(data),
            'zone_cohesion': cls.zone_cohesion(data),
            'space_type_diversity': cls.space_type_diversity(data),
            'vertical_flow_balance': cls.vertical_flow_balance(data),
        }

    @classmethod
    def aggregate(
        cls,
        metrics: Dict[str, float],
        weights: Dict[str, float] = None,
    ) -> float:
        """
        Weighted average of quality metrics.

        Default weights emphasize safety and comfort (1.0) over diversity (0.5).
        Returns float in [0, 1].
        """
        if weights is None:
            weights = cls.DEFAULT_WEIGHTS

        total_w = 0.0
        total_s = 0.0
        for key, w in weights.items():
            if key in metrics:
                total_w += w
                total_s += w * metrics[key]

        if total_w <= 0:
            return 0.5
        return float(total_s / total_w)
