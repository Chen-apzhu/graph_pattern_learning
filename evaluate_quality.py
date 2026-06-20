"""
Graph Quality Evaluator — 图质量评估器

每个指标对应 task.md 中的一条规范或约束。
评估维度：拓扑完整性 | 消防安全性 | 采光合规性 | 声学隔离度 | 交通效率

Usage: PYTHONIOENCODING=utf-8 PYTHONPATH=src python evaluate_quality.py [dataset_dir]
"""

import sys, os
sys.path.insert(0, 'src')

import json, math
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
import networkx as nx

from utils.enums import RoomType, EdgeCategory, EnvNodeType
from graph.school_graph import SchoolGraphData
from graph.graph_utils import GraphAnalyzer


class QualityEvaluator:
    """
    Evaluates a single school graph or an entire dataset against
    building-code-derived quality metrics. Every metric has:
      - A building code reference (GB50099 / GB50016)
      - A formula (explicit, debuggable)
      - A target range
    """

    def __init__(self, sg: SchoolGraphData):
        self.sg = sg
        self.analyzer = GraphAnalyzer(sg)
        self.metrics: Dict[str, dict] = {}

    # ======================================================================
    # 1. TOPOLOGICAL INTEGRITY (task.md §4 连通性)
    # ======================================================================

    def eval_connectivity(self) -> dict:
        """
        图连通性 — GB50016-2014 §5.5.17
        指标: 物理连通图是否全连通 / 孤立节点数 / 桥边数
        目标: 全连通 (1 component), 0 孤立节点, 桥边数 ≤ 楼梯数

        Note: Includes paths through environment nodes (e.g., room→road→room)
        since entrance halls and cafeterias connect via road access.
        """
        # Build graph including ALL physical_connects (room↔room + room↔env)
        full_G = self.analyzer.nx_graph

        # Filter to only physical_connects edges
        G = nx.Graph()
        for node, attrs in full_G.nodes(data=True):
            G.add_node(node, **attrs)
        for u, v, attrs in full_G.edges(data=True):
            if attrs.get('edge_type') == 'physical_connects':
                G.add_edge(u, v)

        room_nodes = [f"room_{i}" for i in range(self.sg.num_rooms)
                      if f"room_{i}" in G]

        # Count room nodes with degree 0 in this graph
        isolated = [f"room_{i}" for i in range(self.sg.num_rooms)
                    if G.degree(f"room_{i}") == 0]
        n_isolated = len(isolated)

        # Check if room nodes are all in one component (allowing env nodes as bridges)
        # Extract only room nodes from each component
        components = list(nx.connected_components(G))
        room_components = []
        for comp in components:
            rooms_in_comp = [n for n in comp if n.startswith('room_')]
            if rooms_in_comp:
                room_components.append(rooms_in_comp)
        n_components = len(room_components)

        # 桥边（去除后图断开）
        bridges = list(nx.bridges(G))
        n_bridges = len(bridges)

        # 得分
        is_connected = (n_components <= 1 and n_isolated == 0)
        score = 100.0
        if not is_connected:
            score -= 30 * n_components  # 每个额外组件扣30
            score -= 5 * n_isolated     # 每个孤立节点扣5

        return {
            'name': '拓扑连通性',
            'ref': 'GB50016-2014 §5.5.17',
            'value': 'PASS' if is_connected else 'FAIL',
            'detail': f'{n_components} 组件, {n_isolated} 孤立节点, {n_bridges} 桥边',
            'score': max(0, score),
            'target': '1 组件, 0 孤立',
            'n_components': n_components,
            'n_isolated': n_isolated,
            'n_bridges': n_bridges,
        }

    # ======================================================================
    # 2. FIRE SAFETY (task.md §4 消防疏散)
    # ======================================================================

    def eval_fire_safety(self) -> dict:
        """
        消防疏散 — GB50016-2014 §5.5
        指标: 高流量房间的物理连通度数
        公式: 对每个房间 r, if occupancy(r) >= 50 then degree_phys(r) >= 2
        """
        room_x = self.sg.room_features
        phys_ei = self.sg.physical_edges

        # 计算每个房间的物理度数
        degrees = defaultdict(int)
        if phys_ei.numel() > 0:
            for j in range(phys_ei.shape[1]):
                s, d = phys_ei[0, j].item(), phys_ei[1, j].item()
                degrees[s] += 1
                degrees[d] += 1

        violations = []
        total_high_occ = 0
        compliant = 0

        for i in range(self.sg.num_rooms):
            occupancy = room_x[i, 15].item() * 300  # 反归一化
            fire_exits = max(1, int(room_x[i, 26].item() * 4))
            if occupancy >= 50:
                total_high_occ += 1
                deg = degrees.get(i, 0)
                if deg >= fire_exits:
                    compliant += 1
                else:
                    rt_idx = room_x[i, :13].argmax().item()
                    rt_name = list(RoomType)[rt_idx].value
                    violations.append({
                        'room_idx': i,
                        'type': rt_name,
                        'occupancy': int(occupancy),
                        'degree': deg,
                        'required': fire_exits,
                    })

        rate = compliant / total_high_occ if total_high_occ > 0 else 1.0
        passed = (rate >= 0.95)

        return {
            'name': '消防疏散',
            'ref': 'GB50016-2014 §5.5',
            'value': f'{rate:.0%}',
            'detail': f'{compliant}/{total_high_occ} 高流量房间满足双出口',
            'score': rate * 100,
            'target': '≥95%',
            'compliant': compliant,
            'total_high_occ': total_high_occ,
            'top_violations': violations[:5],
            'passed': passed,
        }

    # ======================================================================
    # 3. DAYLIGHT COMPLIANCE (task.md §4 采光)
    # ======================================================================

    def eval_daylight(self) -> dict:
        """
        采光合规 — GB50099-2011 §5.1
        指标: daylight_level >= HIGH 的房间是否有 sight_line 边
        公式: 对每个需采光房间 r, degree_sight(r) >= 1
        """
        room_x = self.sg.room_features
        sight_rr = self.sg.sight_room_edges
        sight_re = self.sg.sight_env_edges

        sight_deg = defaultdict(int)
        for ei in [sight_rr, sight_re]:
            if ei.numel() > 0:
                for j in range(ei.shape[1]):
                    s = ei[0, j].item()
                    sight_deg[s] += 1

        total_high = 0
        compliant = 0
        violations = []

        for i in range(self.sg.num_rooms):
            dl = room_x[i, 16].item() * 4  # denormalize
            if dl >= 3.0:  # HIGH or CRITICAL
                total_high += 1
                deg = sight_deg.get(i, 0)
                if deg >= 1:
                    compliant += 1
                else:
                    rt_idx = room_x[i, :13].argmax().item()
                    violations.append({
                        'room_idx': i,
                        'type': list(RoomType)[rt_idx].value,
                        'sight_degree': deg,
                    })

        rate = compliant / total_high if total_high > 0 else 1.0

        return {
            'name': '天然采光',
            'ref': 'GB50099-2011 §5.1',
            'value': f'{rate:.0%}',
            'detail': f'{compliant}/{total_high} 高采光需求房间有视线连接',
            'score': rate * 100,
            'target': '100%',
            'compliant': compliant,
            'total_high': total_high,
            'violations': violations[:3],
            'passed': (rate >= 0.95),
        }

    # ======================================================================
    # 4. ACOUSTIC SEPARATION (task.md §4 声学)
    # ======================================================================

    def eval_acoustic(self) -> dict:
        """
        声学隔离 — GB50099-2011 §7.3
        指标: noise_gap >= 2 的房间对之间有声学阻断边 或 物理距离≥2
        """
        room_x = self.sg.room_features
        acous_ei = self.sg.acoustic_edges
        phys_ei = self.sg.physical_edges

        # 物理邻接表（BFS用）
        phys_adj = {i: set() for i in range(self.sg.num_rooms)}
        if phys_ei.numel() > 0:
            for j in range(phys_ei.shape[1]):
                s, d = phys_ei[0, j].item(), phys_ei[1, j].item()
                phys_adj[s].add(d)
                phys_adj[d].add(s)

        # 声学邻接表
        acous_adj = {i: set() for i in range(self.sg.num_rooms)}
        if acous_ei.numel() > 0:
            for j in range(acous_ei.shape[1]):
                s, d = acous_ei[0, j].item(), acous_ei[1, j].item()
                acous_adj[s].add(d)
                acous_adj[d].add(s)

        total_pairs = 0
        adequate = 0
        violations = []

        for i in range(self.sg.num_rooms):
            for j in range(i + 1, self.sg.num_rooms):
                ni = int(room_x[i, 17].item() * 4)
                nj = int(room_x[j, 17].item() * 4)
                ti = int(room_x[i, 18].item() * 4)
                tj = int(room_x[j, 18].item() * 4)

                if ni - tj >= 2 or nj - ti >= 2:
                    total_pairs += 1
                    has_acoustic = j in acous_adj.get(i, set())

                    if has_acoustic:
                        adequate += 1
                    else:
                        # BFS 最短路径
                        visited = {i}
                        queue = [(i, 0)]
                        path_dist = 999
                        while queue:
                            node, dist = queue.pop(0)
                            if node == j:
                                path_dist = dist
                                break
                            for nb in phys_adj.get(node, set()):
                                if nb not in visited:
                                    visited.add(nb)
                                    queue.append((nb, dist + 1))

                        if path_dist >= 2:
                            adequate += 1
                        else:
                            vi = list(RoomType)[int(room_x[i, :13].argmax().item())].value
                            vj = list(RoomType)[int(room_x[j, :13].argmax().item())].value
                            violations.append({
                                'pair': f'{vi} ↔ {vj}',
                                'noise_gap': max(ni - tj, nj - ti),
                                'path_dist': path_dist,
                                'has_acoustic_edge': False,
                            })

        rate = adequate / total_pairs if total_pairs > 0 else 1.0

        return {
            'name': '声学隔离',
            'ref': 'GB50099-2011 §7.3',
            'value': f'{rate:.0%}',
            'detail': f'{adequate}/{total_pairs} 噪声冲突对有适当隔离',
            'score': rate * 100,
            'target': '100%',
            'adequate': adequate,
            'total_critical_pairs': total_pairs,
            'violations': violations[:3],
            'passed': (rate >= 0.95),
        }

    # ======================================================================
    # 5. CIRCULATION EFFICIENCY (task.md §3.1 动线代理)
    # ======================================================================

    def eval_circulation(self) -> dict:
        """
        交通效率 — GB50099-2011 §8.2.3
        指标: 走廊面积占比 / 平均路径长度 / 图直径
        """
        room_x = self.sg.room_features

        # 走廊面积占比
        total_area = 0.0
        corridor_area = 0.0
        for i in range(self.sg.num_rooms):
            area = room_x[i, 13].item() * 800  # denormalize
            total_area += area
            rt_idx = room_x[i, :13].argmax().item()
            if list(RoomType)[rt_idx] == RoomType.CORRIDOR:
                corridor_area += area

        corridor_ratio = corridor_area / total_area if total_area > 0 else 0

        # 平均物理路径长度
        G = self.analyzer._build_edge_type_subgraph('physical_connects')
        room_nodes = [f"room_{i}" for i in range(self.sg.num_rooms)
                      if f"room_{i}" in G]

        if len(room_nodes) >= 2:
            largest_cc = max(nx.connected_components(G), key=len)
            sub = G.subgraph(largest_cc)
            try:
                avg_path = nx.average_shortest_path_length(sub)
                diameter = nx.diameter(sub)
            except nx.NetworkXError:
                avg_path = 0.0
                diameter = 0
        else:
            avg_path = 0.0
            diameter = 0

        # 介数中心性最大值（识别瓶颈节点）
        bc = nx.betweenness_centrality(G)
        max_bc = max(bc.values()) if bc else 0.0
        bottleneck_nodes = sorted(
            [(n, v) for n, v in bc.items() if n.startswith('room_')],
            key=lambda x: -x[1]
        )[:3]

        # 得分
        ratio_ok = (0.10 <= corridor_ratio <= 0.30)
        score = 100.0
        if not ratio_ok:
            score -= 30
        if avg_path > 0 and avg_path > 8:
            score -= 10

        return {
            'name': '交通效率',
            'ref': 'GB50099-2011 §8.2.3',
            'value': f'走廊比={corridor_ratio:.1%}',
            'detail': (
                f'走廊比={corridor_ratio:.1%} | '
                f'平均路径={avg_path:.1f} | '
                f'直径={diameter} | '
                f'最大介数中心性={max_bc:.3f}'
            ),
            'score': max(0, score),
            'target': '走廊比 10-30%, 平均路径 ≤ 8',
            'corridor_ratio': corridor_ratio,
            'avg_path_length': avg_path,
            'diameter': diameter,
            'max_betweenness': max_bc,
            'bottleneck_nodes': bottleneck_nodes,
            'passed': ratio_ok,
        }

    # ======================================================================
    # 6. GRAPH DIVERSITY (图多样性)
    # ======================================================================

    def eval_diversity(self) -> dict:
        """
        图结构多样性 — 度分布熵 + 房间类型分布的均衡性
        高熵 = 图结构丰富，低熵 = 过于统一
        """
        G = self.analyzer._build_edge_type_subgraph('physical_connects')
        room_nodes = [f"room_{i}" for i in range(self.sg.num_rooms)
                      if f"room_{i}" in G]

        if not room_nodes:
            return {
                'name': '结构多样性',
                'ref': 'N/A',
                'value': 'N/A',
                'detail': 'No room nodes',
                'score': 0,
                'target': '—',
                'passed': False,
                'entropy': 0,
            }

        # 度分布
        degrees = [G.degree(n) for n in room_nodes]
        deg_counts = Counter(degrees)
        total = len(degrees)
        entropy = -sum((c / total) * math.log(c / total)
                       for c in deg_counts.values()) if total > 0 else 0

        # 聚类系数
        clustering = nx.average_clustering(G)

        return {
            'name': '结构多样性',
            'ref': 'N/A',
            'value': f'熵={entropy:.2f}',
            'detail': (
                f'度分布熵={entropy:.2f} | '
                f'平均聚类系数={clustering:.3f} | '
                f'度范围=[{min(degrees)},{max(degrees)}]'
            ),
            'score': min(100, entropy * 40 + clustering * 50),
            'target': '熵 ≥ 1.5 (越高越多样)',
            'entropy': entropy,
            'avg_clustering': clustering,
            'degree_range': (min(degrees), max(degrees)),
            'passed': (entropy >= 1.0),
        }

    # ======================================================================
    # RUN ALL
    # ======================================================================

    def evaluate_all(self) -> Dict[str, dict]:
        self.metrics = {
            'connectivity': self.eval_connectivity(),
            'fire_safety': self.eval_fire_safety(),
            'daylight': self.eval_daylight(),
            'acoustic': self.eval_acoustic(),
            'circulation': self.eval_circulation(),
            'diversity': self.eval_diversity(),
        }
        return self.metrics

    def overall_score(self) -> float:
        if not self.metrics:
            self.evaluate_all()
        scores = [m['score'] for m in self.metrics.values()]
        return float(np.mean(scores))

    def report(self) -> str:
        if not self.metrics:
            self.evaluate_all()

        overall = self.overall_score()
        grade = ('A' if overall >= 85 else 'B' if overall >= 70
                 else 'C' if overall >= 50 else 'D')

        lines = [
            '=' * 72,
            f'  GRAPH QUALITY REPORT — Score: {overall:.1f}/100 ({grade})',
            '=' * 72,
            '',
            f'  {"指标":<16s} {"规范":<24s} {"结果":<12s} {"目标":<16s}',
            f'  {"-"*16} {"-"*24} {"-"*12} {"-"*16}',
        ]

        for key, m in self.metrics.items():
            status = '✓' if m.get('passed', False) else '✗'
            lines.append(
                f'  [{status}] {m["name"]:<14s} {m["ref"]:<24s} '
                f'{m["value"]:<12s} {m["target"]:<16s}'
            )
            if m.get('detail'):
                lines.append(f'       → {m["detail"]}')

        lines.extend([
            '',
            '=' * 72,
            f'  Overall: {overall:.1f}/100  |  Grade: {grade}',
            f'  A≥85  B≥70  C≥50  D<50',
            '=' * 72,
        ])

        return '\n'.join(lines)


# ==========================================================================
# DATASET-LEVEL EVALUATION
# ==========================================================================

def evaluate_dataset(dataset_dir: str, max_graphs: int = None):
    """
    Evaluate all graphs in a dataset and produce aggregate statistics.
    """
    raw = Path(dataset_dir) / 'raw'
    pt_files = sorted(raw.glob('*.pt'))
    if max_graphs:
        pt_files = pt_files[:max_graphs]

    print(f'Evaluating {len(pt_files)} graphs from {dataset_dir}...')
    print()

    all_scores = []
    all_metrics = defaultdict(list)
    detailed_reports = []

    for i, pt_file in enumerate(pt_files):
        bundle = torch.load(str(pt_file), weights_only=False)
        sg = SchoolGraphData(bundle['hetero_data'])
        meta = bundle['metadata']

        evaluator = QualityEvaluator(sg)
        metrics = evaluator.evaluate_all()
        score = evaluator.overall_score()

        all_scores.append(score)
        for key, m in metrics.items():
            all_metrics[key].append(m)

        if i < 3 or i >= len(pt_files) - 3:
            detailed_reports.append((meta['graph_id'], evaluator))

        if (i + 1) % max(1, len(pt_files) // 10) == 0:
            print(f'  [{i+1}/{len(pt_files)}] avg_score={np.mean(all_scores):.1f}')

    # ── Aggregate Report ──
    print()
    print('=' * 72)
    print('  DATASET-LEVEL QUALITY SUMMARY')
    print('=' * 72)
    print(f'  Graphs evaluated: {len(pt_files)}')
    print(f'  Overall score:    {np.mean(all_scores):.1f} +/- {np.std(all_scores):.1f}')
    print(f'  Score range:      [{np.min(all_scores):.0f}, {np.max(all_scores):.0f}]')
    print()

    # Per-metric aggregate
    print(f'  {"Metric":<22s} {"Pass Rate":<12s} {"Avg Score":<12s} {"Status"}')
    print(f'  {"-"*22} {"-"*12} {"-"*12} {"-"*12}')
    for key in ['connectivity', 'fire_safety', 'daylight', 'acoustic',
                'circulation', 'diversity']:
        metrics_list = all_metrics[key]
        pass_rate = sum(1 for m in metrics_list if m.get('passed', False)) / len(metrics_list)
        avg_score = np.mean([m['score'] for m in metrics_list])
        status = '✓ OK' if pass_rate >= 0.85 else '~ WARN' if pass_rate >= 0.50 else '✗ LOW'

        name = metrics_list[0]['name']
        print(f'  {name:<22s} {pass_rate:>8.0%}     {avg_score:>6.1f}      {status}')

    # Score distribution
    print()
    print(f'  Score Distribution:')
    bins = [(0, 30), (30, 50), (50, 70), (70, 85), (85, 101)]
    labels = ['D (<30)', 'D (30-50)', 'C (50-70)', 'B (70-85)', 'A (85+)']
    for (lo, hi), label in zip(bins, labels):
        count = sum(1 for s in all_scores if lo <= s < hi)
        bar = '#' * max(1, count // max(1, len(pt_files) // 40))
        print(f'    {label:12s}: {count:4d}  {bar}')

    # Top 3 / Bottom 3
    scored_graphs = list(zip(pt_files, all_scores))
    scored_graphs.sort(key=lambda x: x[1], reverse=True)

    print()
    print('  Best graphs:')
    for pt_file, score in scored_graphs[:3]:
        print(f'    [{score:.0f}] {pt_file.stem}')

    print('  Worst graphs:')
    for pt_file, score in scored_graphs[-3:]:
        print(f'    [{score:.0f}] {pt_file.stem}')

    print()
    print('=' * 72)

    return {
        'mean_score': float(np.mean(all_scores)),
        'std_score': float(np.std(all_scores)),
        'min_score': float(np.min(all_scores)),
        'max_score': float(np.max(all_scores)),
        'per_metric': {
            key: {
                'pass_rate': float(sum(1 for m in all_metrics[key]
                                   if m.get('passed', False)) / len(all_metrics[key])),
                'avg_score': float(np.mean([m['score'] for m in all_metrics[key]])),
            }
            for key in all_metrics
        },
    }


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == '__main__':
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else 'outputs/dataset_200'

    # 首先展示一个单图详细报告
    raw = Path(dataset_dir) / 'raw'
    sample_file = sorted(raw.glob('*.pt'))[0]
    bundle = torch.load(str(sample_file), weights_only=False)
    sg = SchoolGraphData(bundle['hetero_data'])

    evaluator = QualityEvaluator(sg)
    evaluator.evaluate_all()
    print(evaluator.report())

    # 然后跑数据集级别评估
    print()
    print()
    results = evaluate_dataset(dataset_dir)
