"""
MCTS Subgraph Search — 蒙特卡洛树搜索子图解释器

Implements SubgraphX-style MCTS for finding minimal subgraphs that
explain a GNN model's quality score prediction.

Algorithm: Monte Carlo Tree Search with UCT
  - State: a subgraph (subset of room nodes + edges)
  - Action: drop a room node, drop an edge, or keep
  - Reward: GNN score change + constraint compliance + sparsity bonus

Reference: Yuan et al., "SubgraphX: Explainable GNN via MCTS" (ICML 2021)
"""

from __future__ import annotations

import math
import random
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import networkx as nx

from utils.enums import RoomType


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class SubgraphState:
    """Represents a subgraph of the original school graph."""
    room_ids: Set[str]           # Room node IDs in this subgraph
    env_ids: Set[str]            # Env node IDs
    edge_keys: Set[tuple]        # (src_id, dst_id, edge_type) tuples
    parent_mask: Optional[dict]  # Edge masks applied to reach this state (for HeteroData)
    num_actions_applied: int = 0
    last_action: str = 'ROOT'

    @property
    def num_nodes(self) -> int:
        return len(self.room_ids)

    @property
    def num_edges(self) -> int:
        return len(self.edge_keys)


@dataclass
class MCTSNode:
    """Node in the MCTS search tree."""
    state: SubgraphState
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    untried_actions: List[str] = field(default_factory=list)

    @property
    def q_value(self) -> float:
        """Average reward."""
        return self.total_reward / max(1, self.visits)

    def uct(self, exploration_constant: float = 1.414) -> float:
        """Upper Confidence Bound for Trees."""
        if self.visits == 0:
            return float('inf')
        exploitation = self.q_value
        exploration = exploration_constant * math.sqrt(
            math.log(max(1, self.parent.visits)) / self.visits
        )
        return exploitation + exploration


# ============================================================================
# MCTS Engine
# ============================================================================

class SubgraphMCTS:
    """
    MCTS-based subgraph search for explaining GNN graph quality scores.

    Usage:
        mcts = SubgraphMCTS(scorer, validator, exploration_constant=1.414)
        best_subgraph, score = mcts.search(hetero_data, n_simulations=200)
    """

    def __init__(
        self,
        scorer,                           # MultiTaskScorer or SchoolGraphScorer
        exploration_constant: float = 1.414,
        max_depth: int = 20,
        w_score: float = 1.0,             # Weight for GNN score change
        w_sparsity: float = 0.3,          # Weight for sparsity bonus
        w_connectivity: float = 2.0,      # Penalty for breaking connectivity
        device: str = 'cpu',
        target_metric: str = 'overall_quality',  # Which task head to explain
    ):
        self.scorer = scorer
        self.c = exploration_constant
        self.max_depth = max_depth
        self.w_score = w_score
        self.w_sparsity = w_sparsity
        self.w_connectivity = w_connectivity
        self.device = torch.device(device)
        self.target_metric = target_metric

        # Will be set during search
        self._baseline_score: float = 0.0
        self._original_data = None
        self._all_rooms: List[str] = []
        self._all_edges: Dict[str, List] = {}  # room_id -> [(neighbor_id, edge_type), ...]

    def search(
        self,
        hetero_data,
        n_simulations: int = 200,
        target_sparsity: float = 0.5,
    ) -> Tuple[SubgraphState, float]:
        """
        Run MCTS to find the best explanatory subgraph.

        Args:
            hetero_data: PyG HeteroData of the full school graph.
            n_simulations: Number of MCTS rollouts.
            target_sparsity: Target ratio of nodes to keep (0.5 = keep half).

        Returns:
            (best_state, best_reward)
        """
        self._original_data = hetero_data

        # Compute baseline score (supports both single-task and multi-task scorers)
        self.scorer.eval()
        with torch.no_grad():
            output = self.scorer(hetero_data.to(self.device))
            if isinstance(output, dict):
                self._baseline_score = output.get(self.target_metric, output.get('overall_quality', 0.5)).item()
            else:
                self._baseline_score = output.item()

        # Build room/edge index from the data
        self._build_index()

        # Create root state (full graph)
        root_state = self._make_full_state()
        root = MCTSNode(state=root_state)

        # Generate untried actions
        self._generate_actions(root, target_sparsity)

        best_state = root_state
        best_reward = 0.0

        for sim in range(n_simulations):
            # Selection
            leaf = self._select(root)

            # Expansion
            if leaf.untried_actions:
                leaf = self._expand(leaf)

            # Simulation
            reward = self._simulate(leaf.state)

            # Backpropagation
            self._backpropagate(leaf, reward)

            if reward > best_reward:
                best_reward = reward
                best_state = leaf.state

        return best_state, best_reward

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self):
        """Build room adjacency index from HeteroData."""
        data = self._original_data
        self._all_rooms = list(data['room'].room_ids)
        self._all_edges = defaultdict(list)

        for et in [
            ('room', 'physical_connects', 'room'),
            ('room', 'acoustic_blocks', 'room'),
            ('room', 'sight_lines', 'room'),
        ]:
            try:
                ei = data[et].edge_index
            except (KeyError, AttributeError):
                continue
            if ei.numel() == 0:
                continue
            for j in range(ei.shape[1]):
                src_idx = ei[0, j].item()
                dst_idx = ei[1, j].item()
                src_id = self._all_rooms[src_idx]
                dst_id = self._all_rooms[dst_idx]
                etype_name = et[1]
                self._all_edges[src_id].append((dst_id, etype_name))
                self._all_edges[dst_id].append((src_id, etype_name))

    def _make_full_state(self) -> SubgraphState:
        """Create state representing the full graph."""
        data = self._original_data
        room_ids = set(self._all_rooms)
        env_ids = set(data['environment'].env_ids) if hasattr(data['environment'], 'env_ids') else set()

        edge_keys = set()
        for rid, neighbors in self._all_edges.items():
            for nid, etype in neighbors:
                key = tuple(sorted([rid, nid])) + (etype,)
                edge_keys.add(key)

        return SubgraphState(
            room_ids=room_ids,
            env_ids=env_ids,
            edge_keys=edge_keys,
            parent_mask=None,
        )

    # ------------------------------------------------------------------
    # Action generation
    # ------------------------------------------------------------------

    def _generate_actions(self, node: MCTSNode, target_sparsity: float):
        """Generate legal actions for this state."""
        actions = []
        state = node.state

        # DROP_NODE: can drop any non-critical room
        critical_types = {RoomType.STAIRCASE.value, RoomType.CORRIDOR.value,
                          RoomType.ENTRANCE_HALL.value}
        droppable = [
            rid for rid in state.room_ids
            if not any(rid.startswith(ct) for ct in critical_types)
        ]
        # Only propose if above target sparsity
        stay_threshold = max(5, int(len(self._all_rooms) * target_sparsity))
        if state.num_nodes > stay_threshold:
            for rid in random.sample(droppable, min(5, len(droppable))):
                actions.append(f'DROP_NODE:{rid}')

        # DROP_EDGE: can drop non-critical edges
        if state.num_edges > state.num_nodes:
            sample_edges = random.sample(
                list(state.edge_keys), min(5, len(state.edge_keys))
            )
            for ek in sample_edges:
                actions.append(f'DROP_EDGE:{ek[0]}:{ek[1]}:{ek[2]}')

        # KEEP
        actions.append('KEEP')

        node.untried_actions = actions

    def _apply_action(self, state: SubgraphState, action: str) -> SubgraphState:
        """Apply an action and return the new state."""
        new_rooms = set(state.room_ids)
        new_envs = set(state.env_ids)
        new_edges = set(state.edge_keys)
        desc = action

        if action.startswith('DROP_NODE:'):
            rid = action.split(':', 1)[1]
            new_rooms.discard(rid)
            # Remove all edges incident to this room
            new_edges = {
                ek for ek in new_edges
                if ek[0] != rid and ek[1] != rid
            }
            desc = f'drop {rid}'

        elif action.startswith('DROP_EDGE:'):
            parts = action.split(':')
            a, b, etype = parts[1], parts[2], parts[3]
            key = tuple(sorted([a, b])) + (etype,)
            new_edges.discard(key)
            desc = f'drop edge {a}-{b} ({etype})'

        # KEEP: no change
        return SubgraphState(
            room_ids=new_rooms,
            env_ids=new_envs,
            edge_keys=new_edges,
            parent_mask=state.parent_mask,
            num_actions_applied=state.num_actions_applied + 1,
            last_action=desc,
        )

    # ------------------------------------------------------------------
    # MCTS phases
    # ------------------------------------------------------------------

    def _select(self, root: MCTSNode) -> MCTSNode:
        """Select a leaf node using UCT."""
        node = root
        while node.children:
            # Pick child with highest UCT
            if not node.children:
                break
            node = max(node.children, key=lambda c: c.uct(self.c))
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand by applying one untried action."""
        if not node.untried_actions:
            return node

        action = node.untried_actions.pop(0)
        new_state = self._apply_action(node.state, action)
        child = MCTSNode(state=new_state, parent=node)
        node.children.append(child)

        # Generate actions for the child (if not at max depth)
        if new_state.num_actions_applied < self.max_depth:
            target = max(0.3, 0.5 - new_state.num_actions_applied * 0.02)
            self._generate_actions(child, target)

        return child

    def _simulate(self, state: SubgraphState) -> float:
        """Simulate a rollout: build masked HeteroData and score it."""
        # Build masks from the subgraph state
        try:
            masked_data = self._build_masked_data(state)
        except Exception:
            return -1.0

        # Score the masked graph (supports both single and multi-task)
        self.scorer.eval()
        with torch.no_grad():
            try:
                output = self.scorer(masked_data.to(self.device))
                if isinstance(output, dict):
                    sub_score = output.get(self.target_metric, output.get('overall_quality', 0.5)).item()
                else:
                    sub_score = output.item()
            except Exception:
                return -1.0

        # Reward components
        delta_score = sub_score - self._baseline_score
        sparsity = 1.0 / (1 + state.num_nodes)
        connectivity_penalty = self._check_connectivity(state)

        reward = (
            self.w_score * delta_score
            + self.w_sparsity * sparsity
            - self.w_connectivity * connectivity_penalty
        )
        return reward

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate reward up the tree."""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    # ------------------------------------------------------------------
    # Masked data construction
    # ------------------------------------------------------------------

    def _build_masked_data(self, state: SubgraphState):
        """
        Build a HeteroData with only the nodes/edges in this subgraph.
        """
        data = self._original_data
        N_orig = data['room'].num_nodes

        # Room node mask
        room_mask = torch.zeros(N_orig, dtype=torch.bool)
        for i, rid in enumerate(self._all_rooms):
            if rid in state.room_ids:
                room_mask[i] = True

        # Build edge masks per type
        edge_masks = {}
        for et in [
            ('room', 'physical_connects', 'room'),
            ('room', 'acoustic_blocks', 'room'),
            ('room', 'sight_lines', 'room'),
        ]:
            try:
                ei = data[et].edge_index
            except (KeyError, AttributeError):
                continue
            if ei.numel() == 0:
                continue
            mask = torch.zeros(ei.shape[1], dtype=torch.bool)
            for j in range(ei.shape[1]):
                s = ei[0, j].item()
                d = ei[1, j].item()
                sid = self._all_rooms[s] if s < len(self._all_rooms) else ''
                did = self._all_rooms[d] if d < len(self._all_rooms) else ''
                key = tuple(sorted([sid, did])) + (et[1],)
                if key in state.edge_keys:
                    mask[j] = True
            if mask.any():
                edge_masks[et] = mask

        # Create masked copy of data with room node features zeroed for excluded rooms
        import copy
        masked = copy.copy(data)
        room_x = data['room'].x.clone()
        room_x[~room_mask] = 0.0  # Zero out excluded room features
        masked['room'].x = room_x

        # If no rooms in subgraph, return original (degenerate case)
        if room_mask.sum() == 0:
            return data

        return masked

    # ------------------------------------------------------------------
    # Connectivity check
    # ------------------------------------------------------------------

    def _check_connectivity(self, state: SubgraphState) -> float:
        """
        Check if the physical subgraph is connected.
        Returns 0 if connected, >0 penalty if broken.
        """
        if not state.room_ids:
            return 1.0

        # Build physical adjacency
        phys_adj = defaultdict(set)
        for ek in state.edge_keys:
            a, b, etype = ek
            if etype == 'physical_connects':
                phys_adj[a].add(b)
                phys_adj[b].add(a)

        if not phys_adj:
            return 1.0

        # BFS
        start = next(iter(state.room_ids))
        visited = {start}
        queue = [start]
        while queue:
            node = queue.pop(0)
            for nb in phys_adj.get(node, set()):
                if nb in state.room_ids and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

        connectivity = len(visited) / len(state.room_ids)
        return 1.0 - connectivity  # 0 = fully connected, 1 = fully broken
