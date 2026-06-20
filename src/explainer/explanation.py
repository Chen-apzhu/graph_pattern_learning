"""
Explanation Pipeline — 综合解释报告

Integrates MCTS subgraph search + WL kernel clustering to produce
a comprehensive "Architectural Spatial Pattern Language Dictionary"
(建筑空间模式语言词典).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime

import networkx as nx

from explainer.motif import Motif


def _serialize_graph(graph) -> dict:
    """Serialize a NetworkX graph to JSON-friendly dict for visualization."""
    if graph is None:
        return None
    nodes = []
    for node, attrs in graph.nodes(data=True):
        rid = attrs.get('room_id', str(node))
        rt = rid.split('_')[0] if '_' in rid else 'room'
        nodes.append({
            'id': str(node),
            'room_id': rid,
            'room_type': rt,
        })
    edges = []
    for u, v, attrs in graph.edges(data=True):
        edges.append({
            'source': str(u),
            'target': str(v),
            'edge_type': attrs.get('edge_type', 'physical_connects'),
        })
    return {'nodes': nodes, 'edges': edges}


class ExplanationReport:
    """
    Generates a comprehensive motif dictionary report from clustered subgraphs.

    Output format:
      - JSON: machine-readable motif data
      - Text: human-readable Chinese pattern language dictionary
    """

    def __init__(self, motifs: List[Motif], metadata: dict = None):
        self.motifs = motifs
        self.metadata = metadata or {}
        self.generated_at = datetime.now().isoformat()

    def to_json(self) -> dict:
        """Export motifs as JSON-serializable dict."""
        return {
            'generated_at': self.generated_at,
            'metadata': self.metadata,
            'num_motifs': len(self.motifs),
            'motifs': [
                {
                    'motif_id': m.motif_id,
                    'name': m.name,
                    'room_composition': m.room_composition,
                    'edge_composition': m.edge_composition,
                    'frequency': m.frequency,
                    'percentage': round(m.percentage, 4),
                    'avg_nodes': round(m.avg_nodes, 1),
                    'related_constraints': m.related_constraints,
                    'description': m.description,
                    'centroid_graph': _serialize_graph(m.centroid_graph),
                }
                for m in self.motifs
            ],
        }

    def to_text(self) -> str:
        """Generate the full Chinese pattern language dictionary."""
        lines = [
            "=" * 70,
            "  建筑空间模式语言词典",
            "  Architectural Spatial Pattern Language Dictionary",
            "=" * 70,
            "",
            f"  生成时间: {self.generated_at}",
            f"  发现模体数: {len(self.motifs)}",
            "",
        ]

        if self.metadata:
            lines.append(f"  数据集: {self.metadata.get('dataset', 'N/A')}")
            lines.append(f"  提取子图数: {self.metadata.get('num_subgraphs', 'N/A')}")
            lines.append(f"  聚类数: {self.metadata.get('n_clusters', 'N/A')}")
            lines.append("")

        lines.append("━" * 70)
        lines.append("  模体总览 (按频率排序)")
        lines.append("━" * 70)
        lines.append("")

        for m in self.motifs:
            lines.append(f"  {m.summary()}")

        lines.append("")
        lines.append("━" * 70)
        lines.append("  模体详细描述")
        lines.append("━" * 70)
        lines.append("")

        for m in self.motifs:
            lines.append(m.full_description())

        lines.append("=" * 70)
        lines.append("  词典结束")
        lines.append("=" * 70)

        return '\n'.join(lines)

    def save(self, output_dir: str = 'outputs/explainer'):
        """Save JSON and text reports."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = out / 'motif_dictionary.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_json(), f, indent=2, ensure_ascii=False)

        # Text
        txt_path = out / 'motif_dictionary.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(self.to_text())

        print(f"Report saved to:")
        print(f"  {json_path}")
        print(f"  {txt_path}")

        return str(json_path), str(txt_path)
