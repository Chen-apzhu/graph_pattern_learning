"""
Counterfactual Validator - validates subgraph motifs via counterfactual tests.

Uses MCTS search directly: the MCTS reward (= sub_score - baseline_score)
quantifies how much a subgraph contributes to quality. A subgraph whose
removal DROPS the score significantly is a POSITIVE pattern.

Reference: PLAN.md section 6.4, 9.2
"""

import json, time
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import numpy as np
import networkx as nx

from explainer.mcts_search import SubgraphMCTS


@dataclass
class CFR:
    """Single counterfactual result."""
    test_type: str
    source_graph_id: str
    target_metric: str
    baseline_score: float
    sub_score: float
    delta_q: float
    subgraph_nodes: int
    subgraph_edges: int
    significant: bool

    def to_dict(self):
        return {
            'test_type': self.test_type, 'source_graph_id': self.source_graph_id,
            'target_metric': self.target_metric,
            'baseline_score': round(self.baseline_score, 4),
            'sub_score': round(self.sub_score, 4),
            'delta_q': round(self.delta_q, 4),
            'subgraph_nodes': self.subgraph_nodes,
            'subgraph_edges': self.subgraph_edges,
            'significant': self.significant,
        }


@dataclass
class CFReport:
    """Aggregate counterfactual report."""
    results: List[CFR] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def summary(self):
        if not self.results:
            return {}
        sig = [r for r in self.results if r.significant]
        deltas = [r.delta_q for r in self.results]
        return {
            'total': len(self.results),
            'significant': len(sig),
            'sig_pct': round(len(sig)/len(self.results)*100, 1),
            'mean_delta': round(float(np.mean(deltas)), 4),
            'std_delta': round(float(np.std(deltas)), 4),
            'max_positive': round(float(np.max(deltas)), 4),
        }

    def to_dict(self):
        return {'summary': self.summary(), 'metadata': self.metadata,
                'results': [r.to_dict() for r in self.results]}

    def save(self, path='outputs/explainer/counterfactual_report.json'):
        out = Path(path); out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Counterfactual report: {out}")


class CounterfactualValidator:
    """Validates subgraph importance via MCTS-based removal tests."""

    def __init__(self, scorer, device='cpu', threshold=0.02):
        self.scorer = scorer
        self.device = torch.device(device)
        self.threshold = threshold

    def test_single(
        self, hetero_data, graph_id='unknown', metric='overall_quality', n_sims=50
    ) -> CFR:
        """
        Run MCTS to find the most important subgraph for a given metric.
        delta_q = baseline - sub_score: positive means subgraph IS important.
        """
        mcts = SubgraphMCTS(
            self.scorer, device=str(self.device), target_metric=metric,
            w_score=1.0, w_sparsity=0.0, w_connectivity=0.0,
        )
        try:
            best_state, best_reward = mcts.search(
                hetero_data, n_simulations=n_sims, target_sparsity=0.5
            )
        except Exception as e:
            return CFR('remove', graph_id, metric, 0, 0, 0, 0, 0, False)

        baseline = mcts._baseline_score
        sub_score = best_reward + baseline
        delta = baseline - sub_score

        return CFR(
            test_type='remove', source_graph_id=graph_id,
            target_metric=metric, baseline_score=baseline,
            sub_score=sub_score, delta_q=delta,
            subgraph_nodes=best_state.num_nodes,
            subgraph_edges=best_state.num_edges,
            significant=delta > self.threshold,
        )

    def batch_test(self, dataset_dir, max_graphs=5, metrics=None, n_sims=50) -> CFReport:
        """Run counterfactual tests on a batch of graphs."""
        if metrics is None:
            metrics = ['overall_quality', 'fire_safety_margin', 'circulation_efficiency']

        raw_dir = Path(dataset_dir) / 'raw'
        files = sorted(raw_dir.glob('*.pt'))[:max_graphs]

        report = CFReport(metadata={'dataset': dataset_dir, 'n_sims': n_sims})

        for i, pt_file in enumerate(files):
            bundle = torch.load(str(pt_file), weights_only=False)
            hd = bundle['hetero_data']
            gid = bundle['metadata'].get('graph_id', pt_file.stem)
            print(f"  [{i+1}/{len(files)}] {gid}...")
            for metric in metrics:
                cf = self.test_single(hd, gid, metric, n_sims=n_sims)
                report.results.append(cf)

        return report


if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'src')
    print("Counterfactual Validator loaded.")
    print("Usage: from explainer.counterfactual import CounterfactualValidator")
