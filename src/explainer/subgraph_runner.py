"""
Subgraph Runner — 批量子图提取运行器

Runs MCTS subgraph search on multiple school graphs and collects
subgraph candidates for clustering.

Usage:
    runner = SubgraphRunner(scorer, dataset_dir='outputs/dataset_200_new')
    subgraphs, metadata = runner.run_on_split('test', n_simulations=100)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

import torch
import networkx as nx

from graph.school_graph import SchoolGraphData
from explainer.mcts_search import SubgraphMCTS


class SubgraphRunner:
    """
    Batch runner for MCTS subgraph extraction.

    Args:
        scorer: Trained SchoolGraphScorer instance.
        dataset_dir: Path to dataset.
        device: Torch device.
    """

    def __init__(
        self,
        scorer,
        dataset_dir: str = 'outputs/dataset_200_new',
        device: str = 'cpu',
    ):
        self.scorer = scorer
        self.dataset_dir = Path(dataset_dir)
        self.device = device
        self.mcts = SubgraphMCTS(scorer, device=device)

    def run_on_split(
        self,
        split: str = 'test',
        n_simulations: int = 100,
        max_graphs: int = None,
        target_metrics: list = None,
    ) -> Tuple[List[nx.Graph], List[dict]]:
        """
        Run MCTS on each graph in a dataset split.

        Args:
            split: 'train', 'val', or 'test'.
            n_simulations: MCTS simulations per graph per metric.
            max_graphs: Limit number of graphs (None = all).
            target_metrics: List of metric names to explain (default: ['overall_quality']).

        Returns:
            (subgraphs, metadata) where subgraphs is a list of NetworkX graphs
            and metadata contains per-subgraph info.
        """
        if target_metrics is None:
            target_metrics = ['overall_quality']
        split_dir = self.dataset_dir / split
        pt_files = sorted(split_dir.glob('*.pt'))
        if max_graphs:
            pt_files = pt_files[:max_graphs]

        print(f"Running MCTS on {len(pt_files)} graphs from {split} split...")
        print(f"  Target metrics: {target_metrics}")
        print(f"  Simulations per graph per metric: {n_simulations}")

        all_subgraphs: List[nx.Graph] = []
        all_metadata: List[dict] = []

        t_start = time.time()
        for i, pt_file in enumerate(pt_files):
            bundle = torch.load(str(pt_file), weights_only=False)
            hetero_data = bundle['hetero_data']
            meta = bundle.get('metadata', {})

            sg = SchoolGraphData(hetero_data)
            full_nx = sg.to_networkx()

            for metric in target_metrics:
                self.mcts.target_metric = metric
                try:
                    best_state, best_reward = self.mcts.search(
                        hetero_data,
                        n_simulations=n_simulations,
                        target_sparsity=0.4,
                    )
                except Exception as e:
                    print(f"  [WARN] MCTS failed on {pt_file.name}/{metric}: {e}")
                    continue

                sub_nx = self._state_to_nx(best_state, full_nx)

                if sub_nx.number_of_nodes() > 0:
                    all_subgraphs.append(sub_nx)
                    all_metadata.append({
                        'source_graph': meta.get('graph_id', pt_file.stem),
                        'school_size': meta.get('school_size', 'unknown'),
                        'target_metric': metric,
                        'reward': best_reward,
                        'subgraph_nodes': best_state.num_nodes,
                        'subgraph_edges': best_state.num_edges,
                        'baseline_score': self.mcts._baseline_score,
                        'num_actions': best_state.num_actions_applied,
                    })

            if (i + 1) % max(1, len(pt_files) // 5) == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{i+1}/{len(pt_files)}] {rate:.1f} graphs/s")

        elapsed = time.time() - t_start
        print(f"\nExtracted {len(all_subgraphs)} subgraphs from {len(pt_files)} graphs "
              f"in {elapsed:.1f}s")

        return all_subgraphs, all_metadata

    def _state_to_nx(
        self,
        state,
        full_nx: nx.Graph,
    ) -> nx.Graph:
        """Convert a SubgraphState to a NetworkX graph."""
        sub = nx.Graph()

        # Add nodes
        for node in full_nx.nodes():
            rid = full_nx.nodes[node].get('room_id', node)
            eid = full_nx.nodes[node].get('env_id', '')
            if rid in state.room_ids or eid in state.env_ids:
                sub.add_node(node, **full_nx.nodes[node])

        # Add edges
        for u, v, attrs in full_nx.edges(data=True):
            ru = full_nx.nodes[u].get('room_id', u)
            rv = full_nx.nodes[v].get('room_id', v)
            etype = attrs.get('edge_type', '')
            key = tuple(sorted([ru, rv])) + (etype,)
            if key in state.edge_keys:
                sub.add_edge(u, v, **attrs)

        return sub
