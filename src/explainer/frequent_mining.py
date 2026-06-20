"""
Frequent Subgraph Mining — 频繁子图挖掘

Mines recurring architectural patterns from MCTS-extracted subgraphs
using a simplified gSpan-style approach.

Algorithm:
  1. Label each node by room type (from room_id prefix)
  2. Enumerate all connected k-node subgraphs per graph
  3. Canonicalize via sorted adjacency strings
  4. Count frequency across the dataset
  5. Rank by frequency × avg_score

This is a simplified gSpan — full gSpan requires DFS code canonicalization
which is overkill for our labeled graphs with <100 nodes.
"""

from __future__ import annotations

import json, hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
from itertools import combinations

import networkx as nx
import numpy as np


def mine_frequent_subgraphs(
    graphs: List[nx.Graph],
    metadata: List[dict] = None,
    min_support: int = 3,
    max_nodes: int = 8,
    top_k: int = 15,
) -> List[dict]:
    """
    Mine frequent subgraphs from a collection of NetworkX graphs.

    Args:
        graphs: List of NX graphs (subgraphs extracted by MCTS).
        metadata: Per-graph metadata (reward, score, etc.).
        min_support: Minimum number of graphs a pattern must appear in.
        max_nodes: Maximum subgraph size to enumerate.
        top_k: Number of top patterns to return.

    Returns:
        List of pattern dicts with room_composition, frequency, avg_score, etc.
    """
    if not graphs:
        return []

    n_total = len(graphs)
    print(f'Mining {n_total} graphs (min_support={min_support}, max_nodes={max_nodes})...')

    # Step 1: Label each graph's nodes by room type
    labeled_graphs = []
    for g in graphs:
        lg = _label_graph(g)
        if lg.number_of_nodes() > 0:
            labeled_graphs.append(lg)

    # Step 2: Enumerate subgraphs and count
    pattern_counts: Dict[str, dict] = defaultdict(lambda: {
        'count': 0, 'graphs': set(), 'scores': [],
        'room_comp': {}, 'edge_comp': {},
    })

    for g_idx, g in enumerate(labeled_graphs):
        if g_idx % max(1, n_total // 10) == 0:
            print(f'  Processing graph {g_idx+1}/{n_total}...')

        # Enumerate connected subgraphs up to max_nodes
        subgraphs = _enumerate_connected_subgraphs(g, max_nodes)

        seen_in_this_graph: Set[str] = set()
        for sg in subgraphs:
            canonical = _canonicalize(sg)
            if canonical in seen_in_this_graph:
                continue
            seen_in_this_graph.add(canonical)

            info = pattern_counts[canonical]
            info['count'] += 1
            info['graphs'].add(g_idx)
            if metadata and g_idx < len(metadata):
                info['scores'].append(metadata[g_idx].get('reward', 0))
            # Average room composition
            room_comp = Counter(_node_label(sg, n) for n in sg.nodes())
            for rt, c in room_comp.items():
                prev = info['room_comp'].get(rt, [])
                prev.append(c)
                info['room_comp'][rt] = prev
            edge_comp = Counter(
                sg.edges[u, v].get('edge_type', 'physical_connects')
                for u, v in sg.edges()
            )
            for et, c in edge_comp.items():
                prev = info['edge_comp'].get(et, [])
                prev.append(c)
                info['edge_comp'][et] = prev

    # Step 3: Filter by minimum support
    frequent = []
    for canonical, info in pattern_counts.items():
        support = len(info['graphs'])
        if support < min_support:
            continue
        if info['count'] < min_support * 2:
            continue

        avg_room = {rt: float(np.mean(counts)) for rt, counts in info['room_comp'].items()}
        avg_edge = {et: float(np.mean(counts)) for et, counts in info['edge_comp'].items()}
        avg_score = float(np.mean(info['scores'])) if info['scores'] else 0.0

        # Score: frequency * avg_score * diversity
        diversity = len(avg_room)
        rank = support * (1.0 + avg_score) * (1.0 + 0.1 * diversity)

        frequent.append({
            'canonical': canonical,
            'support': support,
            'total_occurrences': info['count'],
            'percentage': support / n_total,
            'room_composition': avg_room,
            'edge_composition': avg_edge,
            'avg_score': round(avg_score, 3),
            'diversity': diversity,
            'rank': round(rank, 2),
        })

    # Step 4: Sort and deduplicate (remove near-duplicates)
    frequent.sort(key=lambda x: -x['rank'])

    # Remove patterns where one is a subset of another
    deduped = _deduplicate_patterns(frequent)

    # Step 5: Assign names and IDs
    patterns = []
    for i, pat in enumerate(deduped[:top_k]):
        room_str = ', '.join(
            f'{rt}×{c:.0f}' for rt, c in sorted(pat['room_composition'].items(), key=lambda x: -x[1])
        )
        pat['pattern_id'] = f'PAT_{i+1:02d}'
        pat['name'] = _name_pattern(pat['room_composition'])
        pat['summary'] = f"[{pat['pattern_id']}] {pat['name']}: {room_str}"
        patterns.append(pat)

    print(f'  Found {len(frequent)} raw patterns, {len(patterns)} after dedup')
    return patterns


def _label_graph(g: nx.Graph) -> nx.Graph:
    """Relabel graph nodes with room type strings."""
    lg = nx.Graph()
    for node, attrs in g.nodes(data=True):
        rid = attrs.get('room_id', str(node))
        rt = rid.split('_')[0] if '_' in rid else str(node)
        lg.add_node(node, label=rt, **attrs)
    for u, v, attrs in g.edges(data=True):
        lg.add_edge(u, v, **attrs)
    return lg


def _node_label(g: nx.Graph, node) -> str:
    """Get node label from graph."""
    return g.nodes[node].get('label', str(node))


def _enumerate_connected_subgraphs(g: nx.Graph, max_nodes: int) -> List[nx.Graph]:
    """
    Sample connected induced subgraphs with ≤max_nodes nodes.
    Uses random BFS walks to avoid exponential enumeration.
    """
    import random
    results = []
    n = g.number_of_nodes()
    if n == 0:
        return results

    # Number of samples proportional to graph size
    n_samples = min(200, n * 10)

    for _ in range(n_samples):
        size = random.randint(2, min(max_nodes, n))
        start = random.choice(list(g.nodes()))
        # Random BFS walk to get a connected set of 'size' nodes
        selected = {start}
        frontier = list(g.neighbors(start))
        random.shuffle(frontier)
        while len(selected) < size and frontier:
            node = frontier.pop(0)
            if node not in selected:
                selected.add(node)
                for nb in g.neighbors(node):
                    if nb not in selected and nb not in frontier:
                        frontier.append(nb)
        if len(selected) >= 2:
            sg = g.subgraph(selected).copy()
            if nx.is_connected(sg) and sg.number_of_edges() >= 1:
                results.append(sg)

    return results


def _canonicalize(g: nx.Graph) -> str:
    """
    Create a canonical string representation of a labeled graph.
    Uses sorted adjacency lists hashed to a stable string.
    """
    labels = {n: g.nodes[n].get('label', '?') for n in g.nodes()}
    # Build adjacency strings
    adj_strs = []
    for n in sorted(g.nodes()):
        neighbors = sorted(g.neighbors(n))
        nb_str = ','.join(f'{labels.get(nb,"?")}' for nb in neighbors)
        edge_types = []
        for nb in neighbors:
            etype = g.edges[n, nb].get('edge_type', '?')
            edge_types.append(etype)
        et_str = ','.join(sorted(edge_types))
        adj_strs.append(f'{labels[n]}:{nb_str}:{et_str}')

    full = '|'.join(adj_strs)
    return hashlib.md5(full.encode()).hexdigest()[:16]


def _deduplicate_patterns(patterns: List[dict]) -> List[dict]:
    """Remove patterns that are near-duplicates (same room comp, similar size)."""
    result = []
    seen_comps = []

    for pat in patterns:
        comp = frozenset(
            (rt, round(cnt))
            for rt, cnt in pat['room_composition'].items()
        )
        is_dup = False
        for prev in seen_comps:
            overlap = len(comp & prev) / max(1, len(comp | prev))
            if overlap > 0.7:
                is_dup = True
                break
        if not is_dup:
            seen_comps.append(comp)
            result.append(pat)

    return result


def _name_pattern(room_comp: Dict[str, float]) -> str:
    """Generate a descriptive Chinese name for a pattern."""
    top = sorted(room_comp.items(), key=lambda x: -x[1])[:3]
    top_names = []
    name_map = {
        'classroom': '教室', 'corridor': '走道', 'staircase': '楼梯间',
        'toilet': '卫生间', 'office': '办公', 'teacher_office': '教师办公',
        'special_classroom': '专用教室', 'music_room': '音乐教室',
        'library': '图书馆', 'storage': '储藏',
    }

    has_class = room_comp.get('classroom', 0) >= 1
    has_corr = room_comp.get('corridor', 0) >= 1
    has_stair = room_comp.get('staircase', 0) >= 0.5
    has_toilet = room_comp.get('toilet', 0) >= 0.5
    has_office = room_comp.get('office', 0) >= 1
    has_music = room_comp.get('music_room', 0) >= 0.5
    has_special = room_comp.get('special_classroom', 0) >= 0.5

    if has_class and has_corr and has_stair:
        n = int(room_comp.get('classroom', 0))
        return f'{n}教室教学单元'
    elif has_stair and has_toilet and has_corr:
        return '交通服务核'
    elif has_office and has_corr:
        return '办公走廊区'
    elif has_music and has_class:
        return '动静分区单元'
    elif has_special:
        return '专用教室集群'
    elif has_class:
        n = int(room_comp.get('classroom', 0))
        return f'{n}教室集群'
    else:
        items = [name_map.get(rt, rt) for rt, _ in top]
        return '·'.join(items[:2]) + '组合'
